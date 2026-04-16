import { redirect } from "next/navigation";

/**
 * `/` has no UI of its own — just push the user into the first (and
 * currently only) section.  Server-side redirect so there's no flash.
 */
export default function RootPage() {
  redirect("/corpora");
}
